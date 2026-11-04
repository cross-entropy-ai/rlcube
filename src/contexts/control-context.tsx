'use client';

import { PresetsType } from '@react-three/drei/helpers/environment-assets';
import { RefObject, createContext, useContext, useState } from 'react';

import { RubiksCubeRef } from '@/components/rubiks-cube';

type ControlContextType = {
  showRotationIndicators: boolean;
  setShowRotationIndicators: (showRotationIndicators: boolean) => void;
  cubeRoughness: number;
  setCubeRoughness: (cubeRoughness: number) => void;
  cubeSpeed: number;
  setCubeSpeed: (cubeSpeed: number) => void;
  scrambleLength: number;
  setScrambleLength: (scrambleLength: number) => void;
  background: PresetsType;
  setBackground: (background: PresetsType) => void;
  rubiksCubeRef?: RefObject<RubiksCubeRef | null>;
  setRubiksCubeRef: (rubiksCubeRef: RefObject<RubiksCubeRef | null>) => void;
};

export const ControlContext = createContext<ControlContextType>({
  showRotationIndicators: false,
  setShowRotationIndicators: () => {},
  cubeRoughness: 0.5,
  setCubeRoughness: () => {},
  cubeSpeed: 8,
  setCubeSpeed: () => {},
  scrambleLength: 20,
  setScrambleLength: () => {},
  background: 'sunset',
  setBackground: () => {},
  rubiksCubeRef: undefined,
  setRubiksCubeRef: () => {},
});

export const useControlContext = () => {
  return useContext(ControlContext);
};

export const ControlProvider = ({ children }: { children: React.ReactNode }) => {
  const [showRotationIndicators, setShowRotationIndicators] = useState(false);
  const [cubeRoughness, setCubeRoughness] = useState(0.5);
  const [cubeSpeed, setCubeSpeed] = useState(8);
  const [scrambleLength, setScrambleLength] = useState(20);
  const [background, setBackground] = useState<PresetsType>('sunset');
  const [rubiksCubeRef, setRubiksCubeRef] = useState<RefObject<RubiksCubeRef | null> | undefined>(undefined);

  return (
    <ControlContext.Provider
      value={{
        showRotationIndicators,
        setShowRotationIndicators,
        cubeRoughness,
        setCubeRoughness,
        cubeSpeed,
        setCubeSpeed,
        scrambleLength,
        setScrambleLength,
        background,
        setBackground,
        rubiksCubeRef,
        setRubiksCubeRef,
      }}
    >
      {children}
    </ControlContext.Provider>
  );
};
